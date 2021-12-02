"""
.. _lightly-custom-augmentation-5:

Tutorial 5: Custom Augmentations
==============================================

In this tutorial, we will train a model on chest X-ray images in a self-supervised manner.
In self-supervised learning, X-ray images can pose some problems: They are often more
than eight bits deep which makes them incompatible with certain standard torchvision
transforms such as, for example, random-resized cropping. Additionally, some augmentations
which are often used in self-supervised learning are ineffective on X-ray images.
For example, applying color jitter to an X-ray image with a single color channel
does not make sense.

We will show how to address these problems and how to train a ResNet-18 with MoCo
on a set of 16-bit X-ray images in TIFF format.

The original dataset this tutorial is based on can be found `on Kaggle <https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/overview>`_.
These images are in the DICOM format. For simplicity and efficiency reasons, 
we randomly selected ~4000 images from the above dataset, resized them such that the
maximum of the width and height of each image is no larger than 512, and converted
them to the 16-bit TIFF format. To do so, we used ImageMagick which is preinstalled
on most Linux systems. 

.. code::

    mogrify -path path/to/new/dataset -resize 512x512 -format tiff "*.dicom" 

"""

# %%
# Imports
# -------
#
# Import the Python frameworks we need for this tutorial.
import os
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
import pandas
import copy

from lightly.models.modules.heads import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum
from lightly.models.utils import batch_shuffle
from lightly.models.utils import batch_unshuffle

# %%
# Configuration
# -------------
# Let's set the configuration parameters for our experiments.
# 
# We will use eight workers to fetch the data from disc and a batch size of 128.
# The input size of the images is set to 128. With these settings, the training
# requires 2.5GB of GPU memory.

num_workers = 8
batch_size = 128
input_size = 128
seed = 1
max_epochs = 1

# %%
# Let's set the seed for our experiments

pl.seed_everything(seed)

# %%
# Set the path to our dataset

path_to_data = '/datasets/vinbigdata/train_small'

# %%
# Setup custom data augmentations
# -------------------------------
#
# The key to working with 16-bit X-ray images is to convert them to 8-bit images
# which are compatible with the torchvision augmentations without creating harmful
# artifacts. A good way to do so, is to use histogram normalization as described in
# `this paper <https://arxiv.org/pdf/2101.04909.pdf>`_ about Covid-19 prognosis.
#
# Let's write an augmentation, which takes as input a numpy array with 16-bit input
# depth and returns a histogram normalized 8-bit PIL image.

class HistogramNormalize:
    """Performs histogram normalization on numpy array and returns 8-bit image.

    Code was taken and adapted from Facebook:
    https://github.com/facebookresearch/CovidPrognosis

    """

    def __init__(self, number_bins: int = 256):
        self.number_bins = number_bins

    def __call__(self, image: np.array) -> Image:

        # get image histogram
        image_histogram, bins = np.histogram(
            image.flatten(), self.number_bins, density=True
        )
        cdf = image_histogram.cumsum()  # cumulative distribution function
        cdf = 255 * cdf / cdf[-1]  # normalize

        # use linear interpolation of cdf to find new pixel values
        image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
        return Image.fromarray(image_equalized.reshape(image.shape))

# %%
# Since we can't use color jitter on X-ray images, let's replace it and add some
# Gaussian noise instead. It's easiest to apply this after the image has been
# converted to a PyTorch tensor.

class GaussianNoise:
    """Applies random Gaussian noise to a tensor.

    The intensity of the noise is dependent on the mean of the pixel values.
    See https://arxiv.org/pdf/2101.04909.pdf for more information.

    """

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        mu = sample.mean()
        snr = np.random.randint(low=4, high=8)
        sigma = mu / snr
        noise = torch.normal(torch.zeros(sample.shape), sigma)
        return sample + noise

# %%
# Now that we have implemented our custom augmentations, we can combine them
# with available augmentations from the torchvision library to get to the same
# set of augmentations as used in the aforementioned paper. Make sure, that
# the first augmentation is the histogram normalization, and that the Gaussian
# noise is applied after converting the image to a tensor.
#
# Note that we also transform the image from grayscale to RGB by simply repeating
# the single color channel three times. The reason for this is that our ResNet expects
# a three color channel input. This step can be skipped if a different backbone network
# is used.

# compose the custom augmentations with available augmentations
transform = torchvision.transforms.Compose([
    HistogramNormalize(),
    torchvision.transforms.Grayscale(num_output_channels=3),
    torchvision.transforms.RandomResizedCrop(size=input_size, scale=(0.2, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.GaussianBlur(21),
    torchvision.transforms.ToTensor(),
    GaussianNoise(),
])


# %%
# Let's take a look at what our augmentation pipeline does to an image!
# We plot the original image on the left and two random augmentations on the 
# right.

example_image_name = '55e8e3db7309febee415515d06418171.tiff'
example_image_path = os.path.join(path_to_data, example_image_name)
example_image = np.array(Image.open(example_image_path))

# torch transform returns a 3 x W x H image, we only show one color channel
augmented_image_1 = transform(example_image).numpy()[0]
augmented_image_2 = transform(example_image).numpy()[0]

fig, axs = plt.subplots(1, 3)

axs[0].imshow(example_image)
axs[0].set_axis_off()
axs[0].set_title('Original Image')

axs[1].imshow(augmented_image_1)
axs[1].set_axis_off()

axs[2].imshow(augmented_image_2)
axs[2].set_axis_off()

# %%
# Finally, in order to use the augmentation pipeline we defined for self-supervised
# learning, we need to create a lightly collate function like so:

# create a collate function which performs the random augmentations
collate_fn = lightly.data.BaseCollateFunction(transform)

# %%
# Setup dataset and dataloader
# ------------------------------
#
# We create a dataset which points to the images in the input directory. Since
# the input images are 16 bits deep, we need to overwrite the image loader such 
# that it doesn't convert the images to RGB (and hence to 8-bit) automatically.
#
# .. note:: The `LightlyDataset` uses a torchvision dataset underneath, which in turn uses
#   an image loader which transforms the input image to an 8-bit RGB image. If a 16-bit
#   grayscale image is loaded that way, all pixel values above 255 are simply clamped.
#   Therefore, we overwrite the default image loader with our custom one.

def tiff_loader(f):
    """Loads a 16-bit tiff image and returns it as a numpy array.

    """
    with open(f, 'rb') as f:
        image = Image.open(f)
        return np.array(image)

# create the dataset and overwrite the image loader
dataset_train = lightly.data.LightlyDataset(input_dir=path_to_data)
dataset_train.dataset.loader = tiff_loader

# setup the dataloader for training, make sure to pass the collate function
# from above as an argument
dataloader_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=num_workers
)

# %%
# Create the MoCo model
# -----------------------
# Using the building blocks provided by lightly we can write our MoCo model.
# We implement it as a PyTorch Lightning module. For the criterion, we use
# the NTXentLoss which should always be used with MoCo.
#
# MoCo also requires a memory bank - we set its size to 4096 which is approximately
# the size of the input dataset. The temperature parameter of the loss is set to 0.1.
# This smoothens the cross entropy term in the loss function.
#
# The choice of the optimizer is left to the user. Here, we go with simple stochastic
# gradient descent with momentum.

class MoCoModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
        )

        # The backbone has output dimension 512 which defines
        # also the size of the hidden dimension. We select 128
        # for the output dimension.
        self.projection_head = MoCoProjectionHead(512, 512, 128)

        # add the momentum network
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        # create our loss with the memory bank
        self.criterion = lightly.loss.NTXentLoss(
            temperature=0.1, memory_bank_size=4096
        )

    def training_step(self, batch, batch_idx):
        (x_q, x_k), _, _ = batch

        # update momentum
        update_momentum(self.backbone, self.backbone_momentum, 0.99)
        update_momentum(
            self.projection_head, self.projection_head_momentum, 0.99
        )

        # get queries
        q = self.backbone(x_q).flatten(start_dim=1)
        q = self.projection_head(q)

        # get keys
        k, shuffle = batch_shuffle(x_k)
        k = self.backbone_momentum(k).flatten(start_dim=1)
        k = self.projection_head_momentum(k)
        k = batch_unshuffle(k, shuffle)

        loss = self.criterion(q, k)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        # sgd optimizer with momentum
        optim = torch.optim.SGD(
            self.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, max_epochs
        )
        return [optim], [scheduler]


# %%
# Train MoCo with custom augmentations
# -------------------------------------
# Training the self-supervised model is now very easy. We can create a new
# MoCoModel instance and pass it to the PyTorch Lightning trainer.

model = MoCoModel()

gpus = 1 if torch.cuda.is_available() else 0

trainer = pl.Trainer(
    max_epochs=max_epochs,
    gpus=gpus,
    progress_bar_refresh_rate=100,
    precision=16,
)
trainer.fit(model, dataloader_train)


# %%
# Evaluate the results
# ------------------------
# It's always a good idea to evaluate how good the learned representations really
# are. How to do this depends on the available data and metdata. Luckily, in our case,
# we have annotations of critical findings on the X-ray images. We can use this information
# to see, whether images with similar annotations are grouped together.
#
# We start by getting a vector representation of each image in the dataset. For this,
# we create a new dataloader. This time, we can pass the transform directly to the dataset.

# test transforms differ from training transforms as they do not introduce
# additional noise
test_transforms = torchvision.transforms.Compose([
    HistogramNormalize(),
    torchvision.transforms.Grayscale(num_output_channels=3),
    torchvision.transforms.Resize(input_size),
    torchvision.transforms.ToTensor(),
])

# create the dataset and overwrite the image loader as before
dataset_test = lightly.data.LightlyDataset(
    input_dir=path_to_data,
    transform=test_transforms
)
dataset_test.dataset.loader = tiff_loader

# create the test dataloader
dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

# next we add a small helper function to generate embeddings of our images
def generate_embeddings(model, dataloader):
    """Generates representations for all images in the dataloader"""

    embeddings = []
    filenames = []
    with torch.no_grad():
        for img, label, fnames in dataloader:
            img = img.to(model.device)
            emb = model.backbone(img).flatten(start_dim=1)
            embeddings.append(emb)
            filenames.extend(fnames)

    embeddings = torch.cat(embeddings, 0)
    embeddings = normalize(embeddings)
    return embeddings, filenames


# generate the embeddings
# remember to put the model in eval mode!
model.eval()
embeddings, fnames = generate_embeddings(model, dataloader_test)

# %%
# Now, we can use the embeddings to search for nearest neighbors.
#
# We choose three example images. For each example image, we find 50 nearest neighbors.
# Then, we plot the critical findings in the example image (dark blue) and the distribution
# of the critical findings in the nearest neighbor images (light blue) as bar plots.

# transform the original bounding box annotations to multiclass labels
fnames = [fname.split('.')[0] for fname in fnames]

df = pandas.read_csv('/datasets/vinbigdata/train.csv')
classes = list(np.unique(df.class_name))
filenames = list(np.unique(df.image_id))

# iterate over all bounding boxes and add a one-hot label if an image contains
# a bounding box of a given class, after that, the array "multilabels" will 
# contain a row for every image in the input dataset and each row of the 
# array contains a one-hot vector of critical findings for this image
multilabels = np.zeros((len(dataset_test.get_filenames()), len(classes)))
for filename, label in zip(df.image_id, df.class_name):
    try:
        i = fnames.index(filename.split('.')[0])
        j = classes.index(label)
        multilabels[i, j] = 1.
    except Exception:
        pass


def plot_knn_multilabels(
    embeddings, multilabels, samples_idx, filenames, n_neighbors=50
):
    """Plots multiple rows of random images with their nearest neighbors"""
    # lets look at the nearest neighbors for some samples
    # we use the sklearn library
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    _, indices = nbrs.kneighbors(embeddings)

    # position the bars
    bar_width = 0.4
    r1 = np.arange(multilabels.shape[1])
    r2 = r1 + bar_width

    # loop through our randomly picked samples
    for idx in samples_idx:
        fig = plt.figure()
    
        bars1 = multilabels[idx]
        bars2 = np.mean(multilabels[indices[idx]], axis=0)

        plt.title(filenames[idx])
        plt.bar(r1, bars1, color='steelblue', edgecolor='black', width=bar_width)
        plt.bar(r2, bars2, color='lightsteelblue', edgecolor='black', width=bar_width)
        plt.xticks(0.5 * (r1 + r2), classes, rotation=90)
        plt.tight_layout()


# plot the distribution of the multilabels of the k nearest neighbors of 
# the three example images at index 4111, 3340, 1796
k = 20
plot_knn_multilabels(
    embeddings, multilabels, [4111, 3340, 1796], fnames, n_neighbors=k
)
