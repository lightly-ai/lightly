import h5py
import lightly
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision

import math

from PIL import Image, ImageOps
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from tqdm import tqdm

from lightly.data import LightlyDataset, MAECollateFunction
from lightly.models import modules
from lightly.models.modules import heads
from lightly.models.modules import masked_autoencoder
from lightly.models import utils
from lightly.utils import BenchmarkModule
from pytorch_lightning.loggers import TensorBoardLogger

from einops import rearrange

# wandb offline
import os

# os.environ['WANDB_MODE'] = 'offline'

import wandb

wandb.init(
    project="neural_mae_tests",
    entity="maggu",
    settings=wandb.Settings(start_method="thread"),
    save_code=True,
    # config=args,
    # id=args["load_from_wandb"] if args["load_from_wandb"] is not None else None,
    # name=args["wandb_run_name"],
    # resume="must" if args["load_from_wandb"] is not None else False,
)

# set max_epochs to 800 for long run (takes around 10h on a single V100)
max_epochs = 800
knn_k = 200
knn_t = 0.1
classes = 10
original_size = 128
reduction_factor = 4
input_size = original_size // reduction_factor

# benchmark
n_runs = 1  # optional, increase to create multiple runs and report mean + std
batch_size = 4
lr_factor = batch_size / 256  #  scales the learning rate linearly with batch size

# use a GPU if available
gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0


class MAE(nn.Module):
    def __init__(self):
        super().__init__()

        decoder_dim = 512
        vit = torchvision.models.vit_b_32(pretrained=False)

        self.warmup_epochs = 40 if max_epochs >= 800 else 20
        self.mask_ratio = 0.75
        self.patch_size = vit.patch_size
        self.sequence_length = vit.seq_length
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.backbone = masked_autoencoder.MAEBackbone.from_vit(vit)
        self.decoder = masked_autoencoder.MAEDecoder(
            seq_length=vit.seq_length,
            num_layers=1,
            num_heads=16,
            embed_input_dim=vit.hidden_dim,
            hidden_dim=decoder_dim,
            mlp_dim=decoder_dim * 4,
            out_dim=vit.patch_size ** 2 * 3,
            dropout=0,
            attention_dropout=0,
        )
        self.criterion = nn.MSELoss()

    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images, idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(self.mask_token, (batch_size, self.sequence_length))
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode)

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def forward(self, x):
        idx_keep, idx_mask = utils.random_token_mask(
            size=(x.shape[0], self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images, idx_keep)
        x_pred = self.forward_decoder(x_encoded, idx_keep, idx_mask)
        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)

        # rest = utils.get_at_index(patches, idx_keep)

        return x_pred, target

    def training_step(self, batch, batch_idx):
        images, _, _ = batch

        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images, idx_keep)
        x_pred = self.forward_decoder(x_encoded, idx_keep, idx_mask)

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)

        loss = self.criterion(x_pred, target)
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=1.5e-4 * lr_factor,
            weight_decay=0.05,
            betas=(0.9, 0.95),
        )
        cosine_with_warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, self.scale_lr)
        return [optim], [cosine_with_warmup_scheduler]

    def scale_lr(self, epoch):
        if epoch < self.warmup_epochs:
            return epoch / self.warmup_epochs
        else:
            return 0.5 * (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (max_epochs - self.warmup_epochs)))

    def predict(self, images):
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images, idx_keep)
        x_pred = self.forward_decoder(x_encoded, idx_keep, idx_mask)

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)

        # reshape back into image
        x_pred = x_pred.reshape(shape=(target.shape[0], target.shape[1], self.patch_size, self.patch_size, 3))
        target = target.reshape(shape=(target.shape[0], target.shape[1], self.patch_size, self.patch_size, 3))

        x_pred_images = torch.zeros(size=(x_pred.shape[0], images.shape[2], images.shape[3], images.shape[1]))
        target_images = torch.zeros(size=(x_pred.shape[0], images.shape[2], images.shape[3], images.shape[1]))
        for image_id in range(x_pred.shape[0]):
            for idx, orginal_idx in enumerate(idx_mask[image_id]):
                orginal_idx -= 1
                i, j = orginal_idx // (images.shape[2] / self.patch_size), orginal_idx % (
                            images.shape[3] / self.patch_size)
                x_pred_images[image_id, int(i * self.patch_size):int((i + 1) * self.patch_size),
                int(j * self.patch_size):int((j + 1) * self.patch_size), :] = x_pred[image_id, idx, :, :, :]
                target_images[image_id, int(i * self.patch_size):int((i + 1) * self.patch_size),
                int(j * self.patch_size):int((j + 1) * self.patch_size), :] = target[image_id, idx, :, :, :]

        return x_pred_images, target_images


dataset_path = "./mc_rtt_smoothing_output_val.h5"
# load data
with h5py.File(dataset_path, "r") as f:
    data = f["mc_rtt"]
    X_train = np.asarray(data["train_rates_heldin"])
    X_val = np.asarray(data["eval_rates_heldin"])
    y_train = np.array(data["train_rates_heldout"])
    y_val = np.array(data["eval_rates_heldout"])

# X_train = X_train[:, np.newaxis, :, :]
# X_val = X_val[:, np.newaxis, :, :]
# y_train = y_train[:, np.newaxis, :, :]
# y_val = y_val[:, np.newaxis, :, :]
# # copy second dimension to make it 3d
# X_train = np.repeat(X_train, 3, axis=1)
# X_val = np.repeat(X_val, 3, axis=1)
# y_train = np.repeat(y_train, 3, axis=1)
# y_val = np.repeat(y_val, 3, axis=1)
#
# pad_size = 224
# # zero pad dimensions 2 and 3 to make them 128x128
# X_train = np.pad(X_train, ((0, 0), (0, 0), (0, pad_size - X_train.shape[2]), (0, pad_size - X_train.shape[3])), 'constant')
# X_val = np.pad(X_val, ((0, 0), (0, 0), (0, pad_size - X_val.shape[2]), (0, pad_size - X_val.shape[3])), 'constant')

# torch dataset from numpy arrays
# train_dataset = torch.utils.data.TensorDataset(
#     torch.from_numpy(X_train).float(),
#     torch.from_numpy(y_train).float(),
# )
# val_dataset = torch.utils.data.TensorDataset(
#     torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()
# )

path_to_train = './datasets/imagenette2-160/train/'
path_to_test = './datasets/imagenette2-160/val/'

# load each image into a list
train_images = []
train_labels = []
for folder_id, folder in enumerate(os.listdir(path_to_train)):
    if folder == '.DS_Store':
        continue
    for file in os.listdir(path_to_train + folder):
        img_path = path_to_train + folder + '/' + file
        # img = np.array(, dtype='uint8')
        img = Image.open(img_path)
        pad_size = 224
        # pad image using PIL
        img = ImageOps.expand(img, border=(0, 0, pad_size - img.size[0], pad_size - img.size[1]), fill=0)
        img = np.array(img, dtype='uint8')
        # make image 3d
        if img.ndim == 2:
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        train_images.append(img)
        train_labels.append(folder_id)
train_images = np.array(train_images)
train_labels = np.array(train_labels)

train_images = train_images[:100]
train_labels = train_labels[:100]

train_dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(train_images),
    torch.from_numpy(train_labels),
)

vit = torchvision.models.vit_b_16(pretrained=False)
model = MAE()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

dataset = LightlyDataset.from_torch_dataset(train_dataset)

# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")


collate_fn = MAECollateFunction()

dataloader = torch.utils.data.DataLoader(
    # dataset,
    train_dataset,
    batch_size=batch_size,
    # collate_fn=collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=0,
)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4)

print("Starting Training")
for epoch in range(10):
    total_loss = 0
    # for images, _, _ in tqdm(dataloader):
    for images, labels in tqdm(dataloader):
        images = images.to(device)

        images = rearrange(images, 'b h w c -> b c h w')

        # train transform
        images = images[:, :, 0:224, 0:224]
        images = images.float() / 255.
        images = images - 0.5
        images = images * 2

        predictions, targets = model(images)
        loss = criterion(predictions, targets)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print('done')

    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
    # log with wandb
    wandb.log({"loss": avg_loss})

    if epoch % 1 == 0:
        x_pred_images, target_images = model.predict(images)
        vis_images = torch.cat(
            [rearrange(images, 'b c h w -> b h w c'), target_images.to(device), x_pred_images.to(device)], dim=2)
        vis_images /= 2
        vis_images += 0.5
        vis_images *= 255
        vis_images = vis_images[:8]
        vis_images = rearrange(vis_images, 'b h w c -> (b h) w c')
        # to unit8 using torch

        vis_images = vis_images.detach().cpu().numpy()
        wandb.log(
            {
                "mae_image": wandb.Image(
                    vis_images, caption="reconstructions"
                )
            },
            step=epoch,
        )
