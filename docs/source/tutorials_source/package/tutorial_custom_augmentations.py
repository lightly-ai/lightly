"""
.. _lightly-custom-augmentation-5:

Tutorial 5: Custom Augmentations
==============================================

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
from PIL import Image, ImageOps
import numpy as np
import pandas

# %%
# Configuration
# -------------
# TODO
#
num_workers = 8
batch_size = 64
num_splits = 0
seed = 1
max_epochs = 100
num_ftrs = 500

# %%
# Let's set the seed for our experiments
pl.seed_everything(seed)

# %%
# Make sure `path_to_data` points to the downloaded x-ray dataset.
# You can download the dataset `from kaggle <https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/overview>`_.

# %%

path_to_data = '/datasets/vinbigdata/train'


# %%
# TODO

class HistogramEqualize(object):
    """

    """

    def __call__(self, sample: Image):
        """

        """
        return ImageOps.equalize(sample)

class GaussianNoise(object):
    """

    """

    def __call__(self, sample: torch.Tensor):
        """

        """
        mu = sample.mean()
        snr = np.random.randint(low=4, high=8)
        sigma = mu / snr
        noise = torch.normal(torch.zeros(sample.shape), sigma)
        return sample + noise

# %%
# Setup data augmentations and loaders
# ------------------------------------
#
# TODO
transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(size=224, scale=(0.2, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.GaussianBlur(21),
    HistogramEqualize(),
    torchvision.transforms.ToTensor(),
    GaussianNoise(),
])

# %%
#

example_image_name = os.path.join(path_to_data, 'ffeffc54594debf3716d6fcd2402a99f.jpg')
example_image = Image.open(example_image_name)

augmented_image_1 = transform(example_image).numpy()
augmented_image_2 = transform(example_image).numpy()

fig, axs = plt.subplots(1, 3)

axs[0].imshow(np.asarray(example_image))
axs[0].set_axis_off()
axs[0].set_title('Original Image')

axs[1].imshow(augmented_image_1.squeeze())
axs[1].set_axis_off()

axs[2].imshow(augmented_image_2.squeeze())
axs[2].set_axis_off()


# %%
#

collate_fn = lightly.data.BaseCollateFunction(transform)


dataset_train_simclr = lightly.data.LightlyDataset(
    input_dir=path_to_data
)

dataloader_train_simclr = torch.utils.data.DataLoader(
    dataset_train_simclr,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=num_workers
)

# %%
# Create the SimCLR model
# -----------------------
# Create a ResNet backbone and remove the classification head

# TODO
resnet = lightly.models.ResNetGenerator('resnet-18', num_splits=num_splits)
last_conv_channels = list(resnet.children())[-1].in_features
backbone = nn.Sequential(
    *list(resnet.children())[:-1],
    nn.Conv2d(last_conv_channels, num_ftrs, 1),
    nn.AdaptiveAvgPool2d(1)
)

# create the MoCo model using the newly created backbone
model = lightly.models.MoCo(backbone, num_ftrs=num_ftrs, m=0.99)

# TODO
criterion = lightly.loss.NTXentLoss(
    temperature=0.1,
    memory_bank_size=4096,
)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
encoder = lightly.embedding.SelfSupervisedEmbedding(
    model,
    criterion,
    optimizer,
    dataloader_train_simclr
)

gpus = 1 if torch.cuda.is_available() else 0
encoder.train_embedding(gpus=gpus, 
                        #progress_bar_refresh_rate=100,
                        max_epochs=max_epochs,
                        precision=16)


# %%
# What's next?

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    HistogramEqualize(),
    torchvision.transforms.ToTensor(),
])

dataset_test = lightly.data.LightlyDataset(
    input_dir=path_to_data,
    transform=test_transforms
)

dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

encoder.to('cuda')
#embeddings, _, fnames = encoder.embed(dataloader_test, device='cuda')
#embeddings = normalize(embeddings)
embeddings = np.random.randn(len(dataset_test), 5)

df = pandas.read_csv('/datasets/vinbigdata/train.csv')
classes = list(np.unique(df.class_name))
filenames = list(np.unique(df.image_id))

multilabels = np.zeros((len(filenames), len(classes)))
for filename, label in zip(df.image_id, df.class_name):
    i = filenames.index(filename)
    j = classes.index(label)
    multilabels[i, j] = 1.


def plot_knn_multilabels(embeddings, multilabels, n_neighbors=50, num_examples=5):
    """Plots multiple rows of random images with their nearest neighbors
    """
    # lets look at the nearest neighbors for some samples
    # we use the sklearn library
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    _, indices = nbrs.kneighbors(embeddings)

    # get 5 random samples
    samples_idx = np.random.choice(len(indices), size=num_examples, replace=False)

    # TODO
    bar_width = 0.4
    r1 = np.arange(multilabels.shape[1])
    r2 = r1 + bar_width

    # loop through our randomly picked samples
    for idx in samples_idx:
        fig = plt.figure()
    
        bars1 = multilabels[idx]
        bars2 = np.mean(multilabels[indices[idx]], axis=0)
        yer = np.var(multilabels[indices[idx]], axis=0)

        plt.bar(r1, bars1, width=bar_width)
        plt.bar(r2, bars2, width=bar_width, yerr=yer)
        plt.xticks(0.5 * (r1 + r2), classes, rotation=90)
        plt.tight_layout()


plot_knn_multilabels(embeddings, multilabels)
