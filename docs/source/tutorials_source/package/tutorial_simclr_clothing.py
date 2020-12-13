"""

.. _lightly-simclr-tutorial-3:

Tutorial 3: Train SimCLR on Clothing Dataset
==============================================


"""
import os
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import lightly
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import numpy as np


num_workers = 8
batch_size = 256
seed = 1
max_epochs = 2

path_to_data = '/datasets/clothing_dataset/images'

pl.seed_everything(seed)

collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=64,
)

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])

dataset_train_simclr = lightly.data.LightlyDataset(
    from_folder=path_to_data
)

dataset_test = lightly.data.LightlyDataset(
    from_folder=path_to_data,
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

# create a ResNet backbone and remove the classification head
resnet = lightly.models.ResNetGenerator('resnet-18')
last_conv_channels = list(resnet.children())[-1].in_features
backbone = nn.Sequential(
    *list(resnet.children())[:-1],
    nn.Conv2d(last_conv_channels, 32, 1),
    nn.AdaptiveAvgPool2d(1)
)

# %%

model = lightly.models.SimCLR(backbone, num_ftrs=32)

criterion = lightly.loss.NTXentLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

encoder = lightly.embedding.SelfSupervisedEmbedding(
    model,
    criterion,
    optimizer,
    dataloader_train_simclr
)

encoder.train_embedding(gpus=1, max_epochs=max_epochs)

# %%

print('now embedding the data')
embeddings, _, fnames = encoder.embed(dataloader_test)
print('Done...')

# %%
def get_image_as_np_array(filename: str):
    img = Image.open(filename)
    return np.asarray(img)

# %%
# lets look at the nearest neighbors for some samples

nbrs = NearestNeighbors(n_neighbors=3).fit(embeddings)

distances, indices = nbrs.kneighbors(embeddings)

# get 5 random samples
samples_idx = np.random.choice(len(indices), size=5, replace=False)

# loop through our randomly picked samples

for plot_y_offset, idx in enumerate(samples_idx):
    fig = plt.figure()
    # loop through their nearest neighbors
    for plot_x_offset, neighbor_idx in enumerate(indices[idx]):
        ax = fig.add_subplot(1, 3, plot_x_offset + 1)
        fname = os.path.join(path_to_data, fnames[neighbor_idx])
        plt.imshow(get_image_as_np_array(fname))