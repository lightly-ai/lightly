"""
.. _lightly-simclr-tutorial-3:

Tutorial 3: Train SimCLR on Clothing
==============================================

In this tutorial, we will train a SimCLR model using lightly. The model,
augmentations and training procedure is from 
`A Simple Framework for Contrastive Learning of Visual Representations <https://arxiv.org/abs/2002.05709>`_.

The paper explores a rather simple training procedure for contrastive learning.
Since we use the typical contrastive learning loss based on NCE the method
greatly benefits from having larger batch sizes. In this example, we use a batch
size of 256 and paired with the input resolution per image of 64x64 pixels and
a resnet-18 model this example requires 16GB of GPU memory.

We use the 
`clothing dataset from Alex Grigorev <https://github.com/alexeygrigorev/clothing-dataset>`_ 
for this tutorial.

In this tutorial you will learn:

- How to create a SimCLR model

- How different augmentations impact the learned representations

- How to use the SelfSupervisedEmbedding class from the embedding module to train
  a model and obtain embeddings

"""

# %%
# Imports
# -------
#
# Import the Python frameworks we need for this tutorial.
import os
import glob
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import lightly
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from PIL import Image
import numpy as np

# %%
# Configuration
# -------------
# 
# We set some configuration parameters for our experiment.
# Feel free to change them and analyze the effect.
#
# The default configuration with a batch size of 256 and input resolution of 128
# requires 6GB of GPU memory.
num_workers = 8
batch_size = 256
seed = 1
max_epochs = 20
input_size = 128
num_ftrs = 32

# %%
# Let's set the seed for our experiments
pl.seed_everything(seed)

# %%
# Make sure `path_to_data` points to the downloaded clothing dataset.
# You can download it using 
# `git clone https://github.com/alexeygrigorev/clothing-dataset.git`
path_to_data = '/datasets/clothing-dataset/images'


# %%
# Setup data augmentations and loaders
# ------------------------------------
#
# The images from the dataset have been taken from above when the clothing was 
# on a table, bed or floor. Therefore, we can make use of additional augmentations
# such as vertical flip or random rotation (90 degrees). 
# By adding these augmentations we learn our model invariance regarding the 
# orientation of the clothing piece. E.g. we don't care if a shirt is upside down
# but more about the strcture which make it a shirt.
# 
# You can learn more about the different augmentations and learned invariances
# here: :ref:`lightly-advanced`.
collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=input_size,
    vf_prob=0.5,
    rr_prob=0.5
)

# We create a torchvision transformation for embedding the dataset after 
# training
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((input_size, input_size)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])

dataset_train_simclr = lightly.data.LightlyDataset(
    input_dir=path_to_data
)

dataset_test = lightly.data.LightlyDataset(
    input_dir=path_to_data,
    transform=test_transforms
)

dataloader_train_simclr = torch.utils.data.DataLoader(
    dataset_train_simclr,
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
# Create the SimCLR model
# -----------------------
# Create a ResNet backbone and remove the classification head


resnet = torchvision.models.resnet18()
last_conv_channels = list(resnet.children())[-1].in_features
backbone = nn.Sequential(
    *list(resnet.children())[:-1],
    nn.Conv2d(last_conv_channels, num_ftrs, 1),
)

# create the SimCLR model using the newly created backbone
model = lightly.models.SimCLR(backbone, num_ftrs=num_ftrs)

# %%
# We now use the SelfSupervisedEmbedding class from the embedding module.
# First, we create a criterion and an optimizer and then pass them together
# with the model and the dataloader.
criterion = lightly.loss.NTXentLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
encoder = lightly.embedding.SelfSupervisedEmbedding(
    model,
    criterion,
    optimizer,
    dataloader_train_simclr
)

# %% 
# use a GPU if available
gpus = 1 if torch.cuda.is_available() else 0

# %%
# Train the Embedding
# --------------------
# The encoder itself wraps a PyTorch-Lightning module. We can pass any 
# lightning trainer parameter (e.g. gpus=, max_epochs=) to the train_embedding method.
encoder.train_embedding(gpus=gpus, 
                        progress_bar_refresh_rate=100,
                        max_epochs=max_epochs)

# %%
# Now, let's make sure we move the trained model to the gpu if we have one
device = 'cuda' if gpus==1 else 'cpu'
encoder = encoder.to(device)

# %%
# We can use the .embed method to create an embedding of the dataset. The method
# returns a list of embedding vectors as well as a list of filenames.
embeddings, _, fnames = encoder.embed(dataloader_test, device=device)
embeddings = normalize(embeddings)

# %%
# Visualize Nearest Neighbors
#----------------------------
# Let's look at the trained embedding and visualize the nearest neighbors for 
# a few random samples.
#
# We create some helper functions to simplify the work

def get_image_as_np_array(filename: str):
    """Returns an image as an numpy array
    """
    img = Image.open(filename)
    return np.asarray(img)

def plot_knn_examples(embeddings, n_neighbors=3, num_examples=6):
    """Plots multiple rows of random images with their nearest neighbors
    """
    # lets look at the nearest neighbors for some samples
    # we use the sklearn library
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # get 5 random samples
    samples_idx = np.random.choice(len(indices), size=num_examples, replace=False)

    # loop through our randomly picked samples
    for idx in samples_idx:
        fig = plt.figure()
        # loop through their nearest neighbors
        for plot_x_offset, neighbor_idx in enumerate(indices[idx]):
            # add the subplot
            ax = fig.add_subplot(1, len(indices[idx]), plot_x_offset + 1)
            # get the correponding filename for the current index
            fname = os.path.join(path_to_data, fnames[neighbor_idx])
            # plot the image
            plt.imshow(get_image_as_np_array(fname))
            # set the title to the distance of the neighbor
            ax.set_title(f'd={distances[idx][plot_x_offset]:.3f}')
            # let's disable the axis
            plt.axis('off')


# %%
# Let's do the plot of the images. The leftmost image is the query image whereas
# the ones next to it on the same row are the nearest neighbors.
# In the title we see the distance of the neigbor.
plot_knn_examples(embeddings)

# %%
# Color Invariance
# ---------------------
# Let's train again without color augmentation. This will force our model to 
# respect the colors in the images.

# Set color jitter and gray scale probability to 0
new_collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=input_size,
    vf_prob=0.5,
    rr_prob=0.5,
    cj_prob=0.0,
    random_gray_scale=0.0
)

# let's update our collate method and reuse our dataloader
dataloader_train_simclr.collate_fn=new_collate_fn

# create a ResNet backbone and remove the classification head
resnet = torchvision.models.resnet18()
last_conv_channels = list(resnet.children())[-1].in_features
backbone = nn.Sequential(
    *list(resnet.children())[:-1],
    nn.Conv2d(last_conv_channels, num_ftrs, 1),
)
model = lightly.models.SimCLR(backbone, num_ftrs=num_ftrs)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
encoder = lightly.embedding.SelfSupervisedEmbedding(
    model,
    criterion,
    optimizer,
    dataloader_train_simclr
)

encoder.train_embedding(gpus=gpus,
                        progress_bar_refresh_rate=100,
                        max_epochs=max_epochs)
encoder = encoder.to(device)

embeddings, _, fnames = encoder.embed(dataloader_test, device=device)
embeddings = normalize(embeddings)

# %%
# other example
plot_knn_examples(embeddings)

# %%
# What's next?

# You could use the pre-trained model and train a classifier on top.
pretrained_resnet_backbone = model.backbone

# you can also store the backbone and use it in another code
state_dict = {
    'resnet18_parameters': pretrained_resnet_backbone.state_dict()
}
torch.save(state_dict, 'model.pth')

# %%
# THIS COULD BE IN A NEW FILE (e.g. inference.py
#
# Make sure you place the `model.pth` file in the same folder as this code

# load the model in a new file for inference
resnet18_new = torchvision.models.resnet18()
last_conv_channels = list(resnet.children())[-1].in_features
# note that we need to create exactly the same backbone in order to load the weights
backbone_new = nn.Sequential(
    *list(resnet.children())[:-1],
    nn.Conv2d(last_conv_channels, num_ftrs, 1),
)

ckpt = torch.load('model.pth')
backbone_new.load_state_dict(ckpt['resnet18_parameters'])

# %%
# Next Steps
# ------------
#
# Interested in exploring other self-supervised models? Check out our other
# tutorials:
#
# - :ref:`lightly-moco-tutorial-2`
# - :ref:`lightly-simsiam-tutorial-4`
# - :ref:`lightly-custom-augmentation-5`
